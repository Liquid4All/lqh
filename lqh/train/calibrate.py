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
_PROBE_BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


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
) -> bool:
    """POST the measured profile back. Returns True on a 2xx response —
    the standalone calibration job (main) exits non-zero on False so the
    backend resets the row's 'probing' marker."""
    base = _api_base()
    token = os.environ.get("LQH_API_TOKEN")
    if not base or not token:
        return False
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
        r = httpx.post(
            base + "/cloud/batch_profile",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        return 200 <= r.status_code < 300
    except Exception:
        return False


def _apply(training_cfg: dict[str, Any], micro: int, target_effective: int) -> int:
    """Set per_device_batch_size to ``micro`` and adjust grad accumulation
    to preserve at least the configured effective batch size. Returns
    grad_accum."""
    # A micro-batch larger than the requested effective batch silently raises
    # the true optimizer batch because accumulation cannot be below one. This
    # mattered for DPO's target of 16 when a cached/probed GPU profile said
    # 128 or 256 was memory-safe.
    micro = min(max(1, micro), max(1, target_effective))
    accum = max(1, math.ceil(target_effective / micro))
    training_cfg["per_device_batch_size"] = micro
    training_cfg["gradient_accumulation_steps"] = accum
    return accum


def ensure_batch_defaults(
    training_cfg: dict[str, Any],
    *,
    default_micro_batch: int,
    default_effective_batch: int = 256,
) -> None:
    """Fill missing batch knobs with LoRA-friendly defaults.

    ``effective_batch_size`` is a local lqh config field used by the
    auto-tuner as the target optimizer batch. HF Trainer consumes the derived
    ``per_device_batch_size`` and ``gradient_accumulation_steps`` fields.
    Explicit user-provided micro/accum settings are left intact.
    """
    if "per_device_batch_size" not in training_cfg:
        training_cfg["per_device_batch_size"] = default_micro_batch

    micro = max(1, int(training_cfg.get("per_device_batch_size", default_micro_batch)))
    if "gradient_accumulation_steps" not in training_cfg:
        target = int(training_cfg.get("effective_batch_size", default_effective_batch))
        training_cfg["gradient_accumulation_steps"] = max(1, math.ceil(target / micro))

    if "effective_batch_size" not in training_cfg:
        accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
        training_cfg["effective_batch_size"] = max(1, micro * accum)


def _probe_micro_batch(
    model,
    tokenizer,
    *,
    seq_len: int,
    max_micro_batch: int,
    target_headroom: float,
    gradient_checkpointing: bool = True,
    pair_batch: bool = False,
    reserve_bytes: int = 0,
):
    """Probe increasing micro-batches with a real fwd+backward, one
    optimizer step and a short generate(), returning
    (safe_micro_batch, peak_mem_mb) or (None, None).

    Fidelity matters more than speed here (GPU_TYPE_2.md): the caller
    must pass the model in its *training* configuration — PEFT-wrapped
    for LoRA — and we enable gradient checkpointing to match the HF
    Trainer (which turns it on itself, after this probe runs). The
    pre-fix probe measured a full-model backward without checkpointing,
    which inflated memory ~10x for LoRA and made every discovered batch
    far too small.

    ``pair_batch`` shapes the probe for DPO: each candidate b runs 2*b
    sequences (chosen+rejected are concatenated into one forward by
    DPOTrainer) plus an extra no-grad forward at the same width to
    approximate the frozen reference-model pass (same architecture, so
    its activation footprint matches). ``reserve_bytes`` is subtracted
    from the budget for memory the probe cannot see — e.g. the resident
    ref-model weights when calibrating standalone without one loaded.

    Conservative: stops at the first batch that exceeds the memory budget
    or OOMs, and uses the last batch that fit."""
    import torch

    if not torch.cuda.is_available():
        return None, None
    device = next(model.parameters()).device
    if device.type != "cuda":
        return None, None
    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(total * target_headroom) - max(0, int(reserve_bytes))
    if budget <= 0:
        return None, None
    vocab = max(2, int(getattr(tokenizer, "vocab_size", 0) or 32000))

    safe: int | None = None
    peak: int | None = None
    max_micro_batch = max(1, max_micro_batch)
    candidates = [b for b in _PROBE_BATCHES if b <= max_micro_batch]
    if max_micro_batch not in candidates:
        candidates.append(max_micro_batch)
    candidates = sorted(set(candidates))
    was_training = model.training

    # Match the trainer's activation-memory configuration. use_cache must
    # be off under checkpointing (HF disables it the same way); generate()
    # below re-enables it temporarily because decoding without a KV cache
    # is pathological.
    prev_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            # LoRA + checkpointing: the frozen base produces no input
            # grads, so checkpointed blocks need this or backward dies.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        except Exception:
            pass

    for b in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            rows = 2 * b if pair_batch else b
            input_ids = torch.randint(0, vocab, (rows, seq_len), device=device)
            attn = torch.ones_like(input_ids)
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            if pair_batch:
                # Approximate the frozen reference-model forward DPO runs
                # alongside the policy: one no-grad pass at the same pair
                # width (same architecture → comparable activation peak).
                # It must run BEFORE backward — in DPOTrainer the ref
                # logprobs are computed while the policy graph and logits
                # are still live, so the real peak is policy graph + ref
                # forward overlapping. Probing it after backward (which
                # frees the graph) under-counted that peak.
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn)
            out.loss.backward()
            # One zero-LR optimizer step over the trainable params so the
            # Adam state allocation (2 fp32 tensors per trainable param —
            # the dominant term for full fine-tuning, negligible for
            # LoRA) is part of the measurement. lr=0 → weights unchanged.
            opt = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=0.0
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            # Eval-time generate peak — the documented OOM source. Include
            # it so the chosen batch survives checkpoint-eval, not just the
            # train step.
            model.eval()
            with torch.no_grad():
                if gradient_checkpointing:
                    model.config.use_cache = True
                # Generate at the nominal batch (not the pair width):
                # eval/rollout generation batches by example, not by pair.
                prompt = input_ids[:b, : min(seq_len, 32)]
                model.generate(prompt, max_new_tokens=8, do_sample=False)
                if gradient_checkpointing:
                    model.config.use_cache = False
            used = int(torch.cuda.max_memory_reserved(device))
            del opt, input_ids, attn, out
            if used <= budget:
                safe, peak = b, used // (1024 * 1024)
                print(
                    f"calibrate: probe micro_batch={b} peak={used // (1024 * 1024)}MB "
                    f"(budget {budget // (1024 * 1024)}MB)",
                    flush=True,
                )
            else:
                print(
                    f"calibrate: probe micro_batch={b} peak={used // (1024 * 1024)}MB "
                    f"exceeds budget {budget // (1024 * 1024)}MB; stopping",
                    flush=True,
                )
                break
        except Exception as exc:
            # OOM (torch.cuda.OutOfMemoryError) or any probe error — the
            # previous safe batch stands.
            print(f"calibrate: probe micro_batch={b} failed ({type(exc).__name__}); stopping", flush=True)
            break
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if prev_use_cache is not None:
        try:
            model.config.use_cache = prev_use_cache
        except Exception:
            pass
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

    Enabled by ``training.auto_batch`` (default: on). Never raises.
    """
    try:
        enabled = bool(training_cfg.get("auto_batch", True))
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
        target_effective = max(
            1,
            int(training_cfg.get("effective_batch_size", cur_micro * cur_accum)),
        )

        grad_ckpt = bool(training_cfg.get("gradient_checkpointing", True))
        cached = _get_cached_profile(key)
        admin_cap: int | None = None
        if cached:
            try:
                admin_cap = int(cached.get("admin_max_micro_batch") or 0) or None
            except (TypeError, ValueError):
                admin_cap = None
        # The shared cache stores values measured with gradient
        # checkpointing ON (the default; the standalone calibration job
        # probes that way too). A run that disables checkpointing has a
        # far larger activation footprint, so it must not consume — or
        # pollute — those values: it honors the admin cap but otherwise
        # probes locally every run, without write-back.
        if grad_ckpt and cached and cached.get("measured_micro_batch"):
            # Apply the measured value unconditionally (modulo the admin
            # ceiling): it is the ground truth for this exact key. The old
            # `micro >= cur_micro` guard meant a cached value below the
            # config default was ignored and re-probed on every run.
            micro = int(cached["measured_micro_batch"])
            if admin_cap:
                micro = min(micro, admin_cap)
            accum = _apply(training_cfg, micro, target_effective)
            micro = int(training_cfg["per_device_batch_size"])
            print(
                f"calibrate: cached micro_batch={micro} grad_accum={accum} "
                f"(effective {target_effective}, gpu={gpu_type})",
                flush=True,
            )
            return

        # Probe beyond the configured micro-batch up to the requested
        # effective-batch target. The target is a hard ceiling: a micro-batch
        # above it would silently increase the true optimizer batch.
        probe_cap = min(admin_cap or max(_PROBE_BATCHES), target_effective)
        headroom = float(training_cfg.get("batch_headroom", _DEFAULT_HEADROOM))
        safe, peak = _probe_micro_batch(
            model,
            tokenizer,
            seq_len=seq_len,
            max_micro_batch=probe_cap,
            target_headroom=headroom,
            gradient_checkpointing=grad_ckpt,
            pair_batch=method.startswith("dpo"),
        )
        if not safe:
            print("calibrate: probe found no safe batch; keeping configured default", flush=True)
            return
        safe = min(int(safe), probe_cap, target_effective)
        accum = _apply(training_cfg, safe, target_effective)
        if grad_ckpt:
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
        # Same cache contract as maybe_autotune_batch_size: the shared
        # profile stores checkpointing-ON measurements only. An OOM in an
        # uncheckpointed run says nothing about the checkpointed batch,
        # so it must not downgrade the shared value.
        if not bool(training_cfg.get("gradient_checkpointing", True)):
            print("calibrate: OOM downgrade skipped (gradient_checkpointing off — not cached)", flush=True)
            return
        lora_cfg = config.get("lora", {}) or {}
        lora_enabled = bool(lora_cfg.get("enabled", True))
        # Match the profile key the run calibrated under: DPO keys are
        # dpo-prefixed (separate memory shape from SFT — pairs + ref).
        run_type = str(config.get("type", "sft")).lower()
        prefix = "dpo_" if run_type in ("dpo", "on_policy_dpo") else ""
        method = prefix + ("lora" if lora_enabled else "full")
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


# Default LoRA target modules for the standalone probe — keep in sync
# with the sft.py / dpo.py TEXT LoraConfig defaults. The probe builds
# synthetic text-only batches, so it is never run for vision (LFM-VL)
# models — sft.py skips maybe_autotune_batch_size for modality="vision"
# and relies on conservative defaults + the OOM downgrade path (which is
# modality-keyed and works for vision unchanged).
_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "w1", "w2", "w3",
]


def main() -> int:
    """Standalone calibration job: ``python -m lqh.train.calibrate``.

    Launched by the backend when an admin clicks "re-discover" on the
    Model Catalog (KindCalibrate, GPU_TYPE.md §7). The probe key arrives
    via LQH_CALIBRATE_* env; the measurement is POSTed back through the
    scoped job token. Exits non-zero when no safe batch was found or the
    write-back failed, so the backend resets the row's 'probing' marker
    and the job shows as failed instead of silently stale.
    """
    import torch

    base_model = os.environ.get("LQH_CALIBRATE_BASE_MODEL", "").strip()
    if not base_model:
        print("calibrate: LQH_CALIBRATE_BASE_MODEL is required", flush=True)
        return 2
    method = (os.environ.get("LQH_CALIBRATE_METHOD") or "lora").strip().lower()
    if method not in ("lora", "full", "dpo_lora", "dpo_full"):
        print(f"calibrate: unknown method {method!r}", flush=True)
        return 2
    is_lora = method.endswith("lora")
    is_dpo = method.startswith("dpo")
    lora_rank = int(os.environ.get("LQH_CALIBRATE_LORA_RANK") or 0)
    if is_lora and lora_rank <= 0:
        lora_rank = 32
    seq_len = int(os.environ.get("LQH_CALIBRATE_SEQ_LEN") or 2048)
    dtype = (os.environ.get("LQH_CALIBRATE_DTYPE") or "bf16").strip() or "bf16"
    modality = (os.environ.get("LQH_CALIBRATE_MODALITY") or "text").strip() or "text"
    max_micro_env = os.environ.get("LQH_CALIBRATE_MAX_MICRO", "").strip()
    max_micro = int(max_micro_env) if max_micro_env else max(_PROBE_BATCHES)

    if not torch.cuda.is_available():
        print("calibrate: no CUDA device available", flush=True)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    print(
        f"calibrate: loading {base_model} (dtype={dtype}, method={method}, "
        f"lora_rank={lora_rank}, seq_len={seq_len})",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch_dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.02,
            target_modules=_LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # DPO keeps a second frozen copy of the base resident (the reference
    # model). The standalone job doesn't load one, so reserve its weight
    # bytes out of the probe budget instead.
    reserve_bytes = 0
    if is_dpo:
        reserve_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    gpu_type = os.environ.get("LQH_GPU_TYPE") or torch.cuda.get_device_name(0)
    key = profile_key(
        base_model=base_model,
        method=method,
        gpu_type=gpu_type,
        modality=modality,
        seq_len=seq_len,
        lora_rank=lora_rank if is_lora else 0,
        dtype=dtype,
        image_id=os.environ.get("LQH_IMAGE_ID", ""),
    )
    safe, peak = _probe_micro_batch(
        model,
        tokenizer,
        seq_len=seq_len,
        max_micro_batch=max_micro,
        target_headroom=_DEFAULT_HEADROOM,
        gradient_checkpointing=True,
        pair_batch=is_dpo,
        reserve_bytes=reserve_bytes,
    )
    if not safe:
        print("calibrate: probe found no safe micro-batch", flush=True)
        return 1
    # Default effective-batch targets per objective — keep in sync with
    # the submit defaults in lqh/tools/handlers.py and the
    # ensure_batch_defaults calls in sft.py / dpo.py.
    target_effective = {"lora": 256, "dpo_lora": 256, "full": 16, "dpo_full": 2}[method]
    accum = max(1, math.ceil(target_effective / safe))
    if not _post_profile(
        key,
        micro_batch=safe,
        grad_accum=accum,
        peak_mem_mb=peak,
        headroom=_DEFAULT_HEADROOM,
        source="probe",
        stable=True,
    ):
        print("calibrate: failed to write the profile back to the backend", flush=True)
        return 1
    print(
        f"calibrate: done — micro_batch={safe} grad_accum={accum} peak={peak}MB "
        f"(effective {target_effective}, gpu={gpu_type}, model={base_model}, method={method})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
