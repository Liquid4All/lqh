"""In-sandbox LoRA merge for merge-and-serve (INFERENCE.md §2).

``python -m lqh.remote.merge_lora`` runs inside the CPU sandbox the
backend launches for a ``merge_lora`` cloud job (deployments push of a
LoRA artifact). Flow:

1. download the adapter tar from R2 via the presigned
   ``LQH_MERGE_ADAPTER_URL``,
2. load ``LQH_MERGE_BASE_MODEL`` (HF) + adapter with PEFT and
   ``merge_and_unload()`` into full weights,
3. publish the merged checkpoint back through the normal artifact
   register path (``lqh.remote.publish``) with a lineage row marking it
   a *full* checkpoint derived from the base model.

The backend's merge pump then attaches the published checkpoint to the
deployment and drives the pod deploy. CPU-only on purpose: a merge is a
weight addition, memory-bound not compute-bound, and keeping it off
GPUs keeps the one-time merge cost trivial.

Required env (injected by the backend):
    LQH_MERGE_BASE_MODEL    HF repo id of the base model
    LQH_MERGE_ADAPTER_URL   presigned GET for the adapter tar
    LQH_JOB_ID              the merge_lora cloud job id
    LQH_PROJECT_ID          project the merged artifact registers under
    LQH_BASE_URL            api base for artifact registration
    LQH_API_TOKEN           scoped job token (artifacts.write)
Optional:
    HF_TOKEN                for gated base models
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path


def _download(url: str, dest: Path, *, chunk: int = 1 << 20) -> None:
    import httpx

    with httpx.stream("GET", url, timeout=httpx.Timeout(600.0)) as resp:
        if resp.status_code != 200:
            body = resp.read()
            raise RuntimeError(f"adapter download failed ({resp.status_code}): {body[:200]!r}")
        with dest.open("wb") as fh:
            for piece in resp.iter_bytes(chunk):
                fh.write(piece)


def _extract_flat(tar_path: Path, dest: Path) -> Path:
    """Extract and return the directory containing adapter_config.json."""
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")
    for cand in [dest, *sorted(p for p in dest.rglob("*") if p.is_dir())]:
        if (cand / "adapter_config.json").exists():
            return cand
    raise RuntimeError("extracted adapter contains no adapter_config.json")


def _merge(base_model: str, adapter_dir: Path, out_dir: Path) -> None:
    import torch
    from peft import PeftModel

    from lqh.train.load_model import _model_cls, detect_modality

    # Vision (LFM-VL) bases need the image-text-to-text class; text bases
    # keep AutoModelForCausalLM. merge_and_unload() itself is architecture-
    # agnostic (LoRA targets are plain Linears, vision tower included).
    modality = detect_modality(base_model)
    model_cls = _model_cls(modality)

    print(f"merge: loading base {base_model} (bf16, cpu, {modality}) ...", flush=True)
    model = model_cls.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    print("merge: applying adapter ...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()
    print(f"merge: saving merged weights -> {out_dir} ...", flush=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    # The tokenizer (and with it the chat_template — the known LFM
    # failure mode) must travel with the merged checkpoint, otherwise
    # the serving pod renders prompts wrong or not at all. For vision
    # bases the same applies to the image preprocessor config, so the
    # full AutoProcessor is saved instead.
    if modality == "vision":
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(base_model).save_pretrained(str(out_dir))
    else:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))


def main() -> int:
    base_model = os.environ.get("LQH_MERGE_BASE_MODEL", "").strip()
    adapter_url = os.environ.get("LQH_MERGE_ADAPTER_URL", "").strip()
    job_id = os.environ.get("LQH_JOB_ID", "").strip()
    project_id = os.environ.get("LQH_PROJECT_ID", "").strip() or "inference-merge"
    if not base_model or not adapter_url:
        print("merge: LQH_MERGE_BASE_MODEL and LQH_MERGE_ADAPTER_URL are required",
              file=sys.stderr, flush=True)
        return 2

    try:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            tar_path = work / "adapter.tar.gz"
            print("merge: downloading adapter from R2 ...", flush=True)
            _download(adapter_url, tar_path)
            adapter_dir = _extract_flat(tar_path, work / "adapter")

            # publish_run discovers run_dir/model as the final
            # checkpoint candidate; the lineage sidecar marks the result
            # a FULL checkpoint (it is — the adapter is baked in).
            run_dir = work / "run"
            out_dir = run_dir / "model"
            out_dir.mkdir(parents=True)
            _merge(base_model, adapter_dir, out_dir)
            (out_dir / "lineage.json").write_text(json.dumps({
                "artifact_kind": "checkpoint",
                "training_method": "full",
                "base_model": base_model,
            }))

            from lqh.remote.publish import publish_run

            print("merge: publishing merged checkpoint ...", flush=True)
            result = asyncio.run(publish_run(
                run_dir,
                project_id=project_id,
                job_id=job_id or None,
            ))
            if not result.artifacts:
                raise RuntimeError(f"publish failed: {result.failed!r}")
            print(f"merge: done -> artifact {result.artifacts[0].id}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 — terminal status carries the message
        print(f"merge: failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
