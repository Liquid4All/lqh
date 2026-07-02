"""HF checkpoint → GGUF conversion + quantization (GGUF.md).

Two sides live here, mirroring ``lqh.remote.transfer``:

* ``submit_gguf(...)`` — the CLI client. Asks api.lqh.ai to launch a
  CPU-only conversion sandbox (POST /v1/cloud/gguf) and returns the job
  id. Used by the ``gguf_convert`` agent tool.

* ``main()`` / ``python -m lqh.remote.gguf_convert`` — the in-sandbox
  worker. Runs inside the CPU sandbox the backend started:

    1. download the source checkpoint tar from R2 via the presigned
       ``LQH_GGUF_SOURCE_URL`` and extract it,
    2. if it is a LoRA adapter (``LQH_GGUF_IS_LORA``), PEFT-merge it onto
       ``LQH_GGUF_BASE_MODEL`` into full weights first — llama.cpp's
       ``convert_hf_to_gguf.py`` rejects PEFT deltas,
    3. convert the (merged) HF checkpoint to a single f16 GGUF,
    4. quantize the f16 into each requested type (``LQH_GGUF_QUANTS``),
    5. smoke-test every produced GGUF with a short ``llama-cli`` run,
    6. register each ``.gguf`` back as an R2 artifact (kind ``gguf``)
       via the scoped job token, emitting an artifact sentinel per file,
    7. optionally push the files to a Hugging Face repo
       (``LQH_GGUF_HF_REPO``).

  Bytes go R2 → sandbox → (HF) directly; nothing round-trips through the
  laptop or the backend. The llama.cpp toolchain is baked into the
  ``gguf`` Modal image (see backend/scripts/modal_build_image.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# ----------------------------------------------------------------------
# CLI client side
# ----------------------------------------------------------------------


async def submit_gguf(
    *,
    project_id: str,
    source_artifact_id: str,
    quant_types: list[str],
    target_hf_repo: str | None = None,
    private: bool = True,
    include_f16: bool = False,
    base_model: str | None = None,
    artifact_format: str | None = None,
) -> str:
    """Ask the backend to start a CPU GGUF-conversion job. Returns the job id."""
    import httpx

    from lqh.auth import require_token
    from lqh.config import load_config

    config = load_config()
    token = require_token()
    base = config.api_base_url.rstrip("/")
    # api_base_url already ends in /v1 in normal config; tolerate both.
    url = base + ("/cloud/gguf" if base.endswith("/v1") else "/v1/cloud/gguf")

    body: dict[str, object] = {
        "source_artifact_id": source_artifact_id,
        "quant_types": quant_types,
        "include_f16": include_f16,
    }
    if target_hf_repo:
        body["target_hf_repo"] = target_hf_repo
        body["private"] = private
    if base_model:
        body["base_model"] = base_model
    if artifact_format:
        body["artifact_format"] = artifact_format
    # project_id isn't part of the request body (the source artifact
    # carries it) but keep the signature symmetric with submit_transfer
    # in case the backend contract adds it later.
    _ = project_id

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            url, json=body, headers={"Authorization": f"Bearer {token}"}
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"gguf submit failed ({r.status_code}): {r.text[:300]}")
        return r.json()["job_id"]


# ----------------------------------------------------------------------
# In-sandbox launcher side
# ----------------------------------------------------------------------

# Fixed prompt for the post-quantization smoke test. Deliberately a plain
# continuation (not a chat turn) so it works for both base and instruct
# checkpoints without needing the right chat template.
_SMOKE_PROMPT = "The capital of France is"


def _llama_cpp_dir() -> Path:
    """Directory holding convert_hf_to_gguf.py (baked into the image)."""
    return Path(os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp"))


def _bin(name: str, env_key: str) -> str:
    """Resolve a llama.cpp binary: explicit env override, else the build
    dir under LLAMA_CPP_DIR, else the bare name on PATH."""
    if override := os.environ.get(env_key):
        return override
    built = _llama_cpp_dir() / "build" / "bin" / name
    if built.exists():
        return str(built)
    return name


def _run(
    cmd: list[str], *, capture: bool = False, timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command. When ``capture`` is False the
    child inherits stdout/stderr so its logs tee through the launcher.

    stdin is always closed (DEVNULL): llama-cli otherwise blocks waiting
    for interactive input even in non-conversation mode, which would hang
    the whole job until the sandbox timeout. ``timeout`` (seconds) is a
    hard cap so a wedged binary fails fast instead of burning the sandbox
    lease."""
    print(f"gguf: $ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )


def _download(url: str, dest: Path, *, chunk: int = 1 << 20) -> None:
    import httpx

    with httpx.stream("GET", url, timeout=httpx.Timeout(600.0)) as resp:
        if resp.status_code != 200:
            body = resp.read()
            raise RuntimeError(f"R2 download failed ({resp.status_code}): {body[:200]!r}")
        with dest.open("wb") as fh:
            for piece in resp.iter_bytes(chunk):
                fh.write(piece)


def _extract(tar_path: Path, dest: Path) -> None:
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")


def _find_hf_model_dir(root: Path) -> Path:
    """Return the extracted directory that holds a full HF checkpoint
    (config.json + weights)."""
    for cand in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        has_config = (cand / "config.json").exists()
        has_weights = any(
            cand.glob("*.safetensors")
        ) or (cand / "pytorch_model.bin").exists()
        if has_config and has_weights:
            return cand
    raise RuntimeError("extracted checkpoint has no config.json + weights")


def _emit_artifact(artifact_id: str, kind: str) -> None:
    """Emit an artifact sentinel so the SSE stream surfaces the new id
    without the client polling the artifacts list."""
    print(
        "LQH_EVENT_JSON: "
        + json.dumps({"kind": "artifact", "payload": {"artifact_id": artifact_id, "kind": kind}}),
        flush=True,
    )


def _convert_to_f16(model_dir: Path, out_path: Path) -> None:
    convert = _llama_cpp_dir() / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise RuntimeError(f"convert_hf_to_gguf.py not found at {convert}")
    cp = _run([
        sys.executable, str(convert),
        "--outfile", str(out_path),
        "--outtype", "f16",
        str(model_dir),
    ])
    if cp.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"convert_hf_to_gguf.py failed (rc={cp.returncode})")


def _quantize(f16_path: Path, out_path: Path, quant: str) -> None:
    cp = _run([_bin("llama-quantize", "LLAMA_QUANTIZE_BIN"), str(f16_path), str(out_path), quant])
    if cp.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"llama-quantize {quant} failed (rc={cp.returncode})")


def _smoke_test(gguf_path: Path) -> None:
    """Run a short generation and fail if the model errors or produces no
    output beyond the prompt — catches broken conversions before the user
    downloads them (GGUF.md §Testing)."""
    try:
        cp = _run([
            _bin("llama-cli", "LLAMA_CLI_BIN"),
            "-m", str(gguf_path),
            "-p", _SMOKE_PROMPT,
            "-n", "32",
            "--temp", "0",
            "-no-cnv",           # disable conversation mode (stdin is DEVNULL too)
        ], capture=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"smoke test timed out for {gguf_path.name}")
    out = (cp.stdout or "") + (cp.stderr or "")
    if cp.returncode != 0:
        raise RuntimeError(
            f"smoke test failed for {gguf_path.name} (rc={cp.returncode}): {out[-500:]}"
        )
    generated = (cp.stdout or "").replace(_SMOKE_PROMPT, "").strip()
    if not generated:
        raise RuntimeError(f"smoke test produced no output for {gguf_path.name}")
    print(f"gguf: smoke test ok for {gguf_path.name}", flush=True)


def run_gguf(
    *,
    source_url: str,
    filename: str,
    is_lora: bool,
    base_model: str | None,
    quants: list[str],
    include_f16: bool,
    hf_repo: str | None,
    hf_private: bool,
    project_id: str,
    job_id: str | None,
    source_artifact_id: str | None,
) -> None:
    from lqh.artifacts import BackendArtifactStore

    # Work under /workspace (the project volume) when available so large
    # f16 + quant files don't fill the small sandbox-local disk.
    vol = Path("/workspace")
    work = Path(tempfile.mkdtemp(dir=str(vol) if vol.exists() else None, prefix="gguf-"))
    stem = Path(filename or "model").name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem or "model"

    print("gguf: downloading source checkpoint from R2 ...", flush=True)
    tar_path = work / "source.tar.gz"
    _download(source_url, tar_path)
    src_dir = work / "src"
    src_dir.mkdir()
    _extract(tar_path, src_dir)

    # Resolve the HF checkpoint directory to convert.
    if is_lora:
        if not base_model:
            raise RuntimeError("LoRA source requires LQH_GGUF_BASE_MODEL to merge onto")
        from lqh.remote.merge_lora import _extract_flat, _merge

        # _extract_flat locates the adapter dir (adapter_config.json).
        adapter_dir = _extract_flat(tar_path, work / "adapter")
        model_dir = work / "merged"
        model_dir.mkdir()
        print(f"gguf: merging LoRA onto {base_model} ...", flush=True)
        _merge(base_model, adapter_dir, model_dir)
    else:
        model_dir = _find_hf_model_dir(src_dir)

    # Convert once to f16, then quantize into each requested type.
    f16_path = work / f"{stem}-f16.gguf"
    print("gguf: converting to f16 ...", flush=True)
    _convert_to_f16(model_dir, f16_path)

    produced: list[tuple[Path, str]] = []  # (path, quant-label)
    if include_f16:
        produced.append((f16_path, "F16"))
    for quant in quants:
        out_path = work / f"{stem}-{quant.lower()}.gguf"
        print(f"gguf: quantizing -> {quant} ...", flush=True)
        _quantize(f16_path, out_path, quant)
        produced.append((out_path, quant))

    # Smoke-test every produced GGUF before publishing.
    for path, _quant in produced:
        _smoke_test(path)

    # Register each artifact (kind=gguf) via the scoped job token.
    store = BackendArtifactStore(
        api_base=os.environ.get("LQH_BASE_URL"),
        token=os.environ.get("LQH_API_TOKEN"),
    )
    import asyncio

    parent_ids = [source_artifact_id] if source_artifact_id else []

    # Optional HF push target: create the repo once up front so each file
    # can be pushed before we register its artifact row with hf_repo set.
    hf_api = None
    if hf_repo:
        from huggingface_hub import HfApi

        hf_api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        hf_api.create_repo(repo_id=hf_repo, repo_type="model", private=hf_private, exist_ok=True)

    registered = []
    for path, quant in produced:
        # Push to HF first (when requested) so the artifact row only claims
        # an hf_repo once the file is actually there.
        pushed_repo = None
        if hf_api is not None:
            print(f"gguf: uploading {path.name} to hf:{hf_repo} ...", flush=True)
            hf_api.upload_file(
                repo_id=hf_repo,
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_type="model",
            )
            pushed_repo = hf_repo

        # lineage.artifact_kind uses the coarse artifact_lineage enum
        # (migration 0013), which does NOT include "gguf" — "other" is the
        # catch-all. The artifacts.kind column (="gguf") is what identifies
        # these; lineage just records provenance (base + source checkpoint).
        lineage = {
            "artifact_kind": "other",
            "parent_ids": parent_ids,
            "hyperparams": {"quant": quant, "gguf_stem": stem},
        }
        if base_model:
            lineage["base_model"] = base_model
        handle = asyncio.run(store.upload_file(
            path,
            project_id=project_id,
            kind="gguf",
            job_id=job_id,
            lineage=lineage,
            hf_repo=pushed_repo,
        ))
        print(f"gguf: registered {path.name} -> artifact {handle.id}", flush=True)
        _emit_artifact(handle.id, "gguf")
        registered.append((path, handle))

    if hf_repo:
        print(f"gguf: pushed {len(registered)} files -> hf:{hf_repo}", flush=True)
    print(f"gguf: done -> {len(registered)} artifacts", flush=True)


def main(argv: list[str] | None = None) -> int:
    source_url = os.environ.get("LQH_GGUF_SOURCE_URL", "")
    quants_raw = os.environ.get("LQH_GGUF_QUANTS", "")
    quants = [q.strip() for q in quants_raw.split(",") if q.strip()]
    is_lora = os.environ.get("LQH_GGUF_IS_LORA", "0") == "1"
    base_model = os.environ.get("LQH_GGUF_BASE_MODEL", "").strip() or None
    include_f16 = os.environ.get("LQH_GGUF_INCLUDE_F16", "0") == "1"
    hf_repo = os.environ.get("LQH_GGUF_HF_REPO", "").strip() or None
    hf_private = os.environ.get("LQH_GGUF_HF_PRIVATE", "1") != "0"
    filename = os.environ.get("LQH_GGUF_FILENAME", "model.tar.gz")
    project_id = os.environ.get("LQH_PROJECT_ID", "").strip() or "gguf-convert"
    job_id = os.environ.get("LQH_JOB_ID", "").strip() or None
    source_artifact_id = os.environ.get("LQH_GGUF_SOURCE_ARTIFACT_ID", "").strip() or None

    if not source_url or not quants:
        print("gguf: LQH_GGUF_SOURCE_URL and LQH_GGUF_QUANTS are required",
              file=sys.stderr, flush=True)
        return 2
    try:
        run_gguf(
            source_url=source_url,
            filename=filename,
            is_lora=is_lora,
            base_model=base_model,
            quants=quants,
            include_f16=include_f16,
            hf_repo=hf_repo,
            hf_private=hf_private,
            project_id=project_id,
            job_id=job_id,
            source_artifact_id=source_artifact_id,
        )
    except Exception as exc:  # noqa: BLE001 - terminal status carries the message
        print(f"gguf: failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
