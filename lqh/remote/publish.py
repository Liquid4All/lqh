"""End-of-run artifact publisher for remote backends.

Invoked as ``python -m lqh.remote.publish <run_dir>`` on the remote
machine after a successful train/infer run. Walks the run dir,
classifies each output, uploads to R2 via ``api.lqh.ai``, and writes
an ``artifacts.json`` manifest in the run dir that the client picks
up on the next rsync.

This replaces the previous "rsync everything including multi-GB
checkpoints back over SSH" path. Only the manifest + small files
(metrics, status, logs) flow over SSH after this; heavy artifacts
go remote → R2 → client directly via presigned URLs when needed.

Designed to be self-contained: stdlib + httpx, no lqh.train deps.
Auth and API base come from environment variables sourced by the
launcher (lqh/remote/ssh_direct.py writes them into
``.lqh-env/.env``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Self-contained imports: pull only the artifacts module from lqh.
# Avoid importing lqh.train so the remote venv doesn't need torch
# in its publish-only path.
from lqh.artifacts import (
    ArtifactError,
    ArtifactHandle,
    ArtifactKind,
    BackendArtifactStore,
)

logger = logging.getLogger(__name__)

__all__ = ["main", "publish_run", "PublishResult"]


@dataclass
class _Candidate:
    """One thing to publish: either a single file or a directory to be
    tar'd. ``kind`` is the registered artifact kind; ``relpath`` is the
    path relative to the run dir for human-readable manifest entries.

    ``lineage`` is an optional dict matching the ArtifactLineageInput
    schema; when set, it's passed to the register call so the backend
    writes an `artifact_lineage` row alongside the artifact row.
    """

    path: Path
    kind: ArtifactKind
    relpath: str
    is_dir: bool = False
    lineage: dict | None = None


@dataclass
class PublishResult:
    """Returned by ``publish_run`` and written to ``artifacts.json``."""

    artifacts: list[ArtifactHandle]
    failed: list[tuple[str, str]]  # (relpath, error_message)

    def to_json(self) -> dict:
        return {
            "artifacts": [
                {
                    "id": h.id,
                    "kind": h.kind,
                    "size_bytes": h.size_bytes,
                    "r2_key": h.r2_key,
                    "job_id": h.job_id,
                    "sha256": h.sha256,
                }
                for h in self.artifacts
            ],
            "failed": [
                {"relpath": rp, "error": err} for rp, err in self.failed
            ],
        }


def _load_lineage_sidecar(target: Path) -> dict | None:
    """Look for ``<target>.lineage.json`` (file) or
    ``<target>/lineage.json`` (dir). Returns the parsed dict or None.

    The trainer drops these next to each artifact it intends to
    publish: a SFT checkpoint dir gets ``checkpoints/4500/lineage.json``,
    a rollouts parquet gets ``iterations/2/predictions.parquet.lineage.json``.
    The publisher reads them on its sweep so the lineage row is
    written in the same call as the artifact registration.
    """
    if target.is_dir():
        cand = target / "lineage.json"
    else:
        cand = target.with_suffix(target.suffix + ".lineage.json")
    if cand.exists() and cand.is_file():
        try:
            return json.loads(cand.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read %s: %s", cand, exc)
    return None


def _dir_has_weights(d: Path) -> bool:
    """True when ``d`` looks like a real saved checkpoint (i.e. carries
    model weights), False when it's a sidecar dir holding only eval
    outputs.

    SFT writes two distinct shapes under ``checkpoints/``:
      * ``checkpoints/step_N/`` — eval-only dir produced by the
        per-checkpoint eval callback; contains ``predictions.parquet``
        and ``eval_request.json``, no weights.
      * ``checkpoints/checkpoint-N/`` (HF Trainer default) — real
        checkpoint with ``*.safetensors`` or ``adapter_model.*``.

    Tarring an eval-only dir as ``kind=checkpoint`` mislabels it in
    the artifact list and wastes R2 storage. Gating on the presence
    of weight files distinguishes them without coupling to a specific
    naming convention.
    """
    if not d.is_dir():
        return False
    for entry in d.iterdir():
        name = entry.name
        if name.endswith(".safetensors") or name == "pytorch_model.bin":
            return True
        if name.startswith("adapter_model.") or name.startswith("model.safetensors"):
            return True
    return False


def _resolve_candidates(run_dir: Path) -> list[_Candidate]:
    """Enumerate publishable outputs in a run dir.

    Order matters only for stable logging — the upload itself happens
    in this order, which means small files land first and the user
    sees progress before a multi-GB checkpoint tar starts.
    """
    out: list[_Candidate] = []

    # --- run-root small files ---
    smalls: dict[str, ArtifactKind] = {
        "progress.jsonl": "metrics",
        "status.json": "metrics",
        "eval_request.json": "eval_result",
        "eval_result.json": "eval_result",
        "predictions.parquet": "predictions",
        "stdout.log": "logs",
        "stderr.log": "logs",
        "config.json": "other",
        "eval_history.json": "metrics",
    }
    for name, kind in smalls.items():
        p = run_dir / name
        if p.exists() and p.is_file():
            out.append(_Candidate(path=p, kind=kind, relpath=name))

    # --- logs/ — the cloud launcher tees stdout/stderr into
    # run_dir/logs/ (see the cloud runner's buildLauncherScript). The
    # SSH-direct backend writes them at run_dir top-level, so we
    # support both layouts.
    logs_dir = run_dir / "logs"
    if logs_dir.exists() and logs_dir.is_dir():
        for name in ("stdout.log", "stderr.log"):
            p = logs_dir / name
            if p.exists() and p.is_file():
                out.append(_Candidate(path=p, kind="logs", relpath=f"logs/{name}"))

    # --- model/ or model-lora/ — the final dir written by the SFT
    # trainer (run_dir/model/ when merged, run_dir/model-lora/ when
    # saving the LoRA adapter alone). Tar-and-upload as a single
    # checkpoint artifact. The dir name carries forward to the R2
    # tar filename (model.tar.gz vs model-lora.tar.gz) so adapter
    # checkpoints are visually distinguishable in the artifact list
    # without needing a separate ArtifactKind. Without this branch
    # a successful fine-tune leaves no model state in R2.
    for sub in ("model", "model-lora"):
        p = run_dir / sub
        if _dir_has_weights(p):
            out.append(
                _Candidate(
                    path=p,
                    kind="checkpoint",
                    relpath=sub,
                    is_dir=True,
                )
            )

    # --- checkpoints/<step>/ — directory per checkpoint, tar each ---
    # plus surface the eval-side files inside as their own typed
    # artifacts. This mirrors the iterations/ branch below: the whole
    # dir lands as one checkpoint tar (for restore), AND the small
    # eval-related files inside (predictions.parquet from
    # _run_checkpoint_eval, eval_request.json from write_eval_request)
    # are uploaded separately so consumers querying for ``kind=
    # predictions`` or ``kind=eval_result`` get sensible results
    # without having to download and untar the whole checkpoint.
    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.exists() and ckpt_root.is_dir():
        for sub in sorted(ckpt_root.iterdir()):
            if sub.is_dir():
                # Only tar the dir as a checkpoint artifact if it
                # actually carries weights — see _dir_has_weights.
                # Eval-only step_N dirs still surface their inner
                # predictions / eval_result files individually below.
                if _dir_has_weights(sub):
                    out.append(
                        _Candidate(
                            path=sub,
                            kind="checkpoint",
                            relpath=f"checkpoints/{sub.name}",
                            is_dir=True,
                        )
                    )
                for fname, kind in (
                    ("predictions.parquet", "predictions"),
                    ("eval_request.json", "eval_result"),
                    ("eval_result.json", "eval_result"),
                ):
                    p = sub / fname
                    if p.exists() and p.is_file():
                        out.append(
                            _Candidate(
                                path=p,
                                kind=kind,
                                relpath=f"checkpoints/{sub.name}/{fname}",
                            )
                        )
            elif sub.is_file():
                # eg checkpoints/best.json — small file alongside dirs
                # Skip lineage sidecars themselves; they're metadata
                # *about* artifacts, not artifacts in their own right.
                if sub.name.endswith(".lineage.json"):
                    continue
                out.append(
                    _Candidate(
                        path=sub,
                        kind="metrics",
                        relpath=f"checkpoints/{sub.name}",
                    )
                )

    # --- iterations/<i>/ — DPO iter dirs ---
    iter_root = run_dir / "iterations"
    if iter_root.exists() and iter_root.is_dir():
        for sub in sorted(iter_root.iterdir()):
            for fname, kind in (
                ("predictions.parquet", "predictions"),
                ("eval_predictions.parquet", "eval_result"),
                ("iter_request.json", "other"),
            ):
                p = sub / fname
                if p.exists() and p.is_file():
                    out.append(
                        _Candidate(
                            path=p,
                            kind=kind,
                            relpath=f"iterations/{sub.name}/{fname}",
                        )
                    )
            # Per-iteration checkpoint dir (lqh/train/dpo.py writes
            # iter_dir / "checkpoint" via save_pretrained). Gating on
            # _dir_has_weights skips the dir on incomplete iters or if
            # the saver was disabled.
            ckpt_sub = sub / "checkpoint"
            if _dir_has_weights(ckpt_sub):
                out.append(
                    _Candidate(
                        path=ckpt_sub,
                        kind="checkpoint",
                        relpath=f"iterations/{sub.name}/checkpoint",
                        is_dir=True,
                    )
                )

    # --- sweep_<id>/ — hyperparameter-sweep per-config sub-dirs ---
    # Each contains the child trainer's stdout.log, stderr.log,
    # progress.jsonl, eval_history.json, model{,-lora}/, and
    # checkpoints/. Without this branch all that's published from a
    # sweep run is the OUTER sweep's logs — the child's actual
    # training output is invisible, which is the failure mode that
    # made the before/after test impossible to debug.
    for sub in sorted(run_dir.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("sweep_")):
            continue
        # Small files at the sub-dir root.
        sub_smalls: dict[str, ArtifactKind] = {
            "config.json":       "other",
            "stdout.log":        "logs",
            "stderr.log":        "logs",
            "progress.jsonl":    "metrics",
            "eval_history.json": "metrics",
            "chosen_ce_summary.json": "metrics",
            "status.json":       "metrics",
        }
        for fname, kind in sub_smalls.items():
            p = sub / fname
            if p.exists() and p.is_file():
                out.append(
                    _Candidate(
                        path=p,
                        kind=kind,
                        relpath=f"{sub.name}/{fname}",
                    )
                )
        # Per-config model/checkpoints — same shape as the top-level
        # paths, but nested. Saves the trained adapter so a sweep
        # can be inspected after the fact.
        for mdir in ("model", "model-lora"):
            p = sub / mdir
            if _dir_has_weights(p):
                out.append(
                    _Candidate(
                        path=p,
                        kind="checkpoint",
                        relpath=f"{sub.name}/{mdir}",
                        is_dir=True,
                    )
                )

    # Single sweep to attach any lineage sidecar files. Cheap (just a
    # stat per candidate) and keeps the per-block logic above focused
    # on "what to publish" rather than "metadata to attach".
    for c in out:
        c.lineage = _load_lineage_sidecar(c.path)

    return out


def _tar_directory(src: Path, dest: Path) -> None:
    """Create a gzipped tar at ``dest`` containing ``src`` rooted as its
    basename. Streaming, deterministic order, no compression-level
    knob (gzip default keeps a checkpoint at ~95% of original size for
    safetensors which barely compress; the win is one file instead of
    thousands)."""
    with tarfile.open(dest, "w:gz") as tar:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                arcname = f.relative_to(src.parent)
                tar.add(f, arcname=str(arcname))


async def publish_run(
    run_dir: Path,
    *,
    project_id: str,
    job_id: str | None = None,
    api_base: str | None = None,
    token: str | None = None,
) -> PublishResult:
    """Walk ``run_dir`` and upload artifacts. Returns a ``PublishResult``.

    Never raises on a per-file failure — that file ends up in
    ``failed`` instead. Raises only on a programmer error (bad args)
    or an unauthenticated backend.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    store = BackendArtifactStore(api_base=api_base, token=token)
    candidates = _resolve_candidates(run_dir)

    successes: list[ArtifactHandle] = []
    failures: list[tuple[str, str]] = []

    for cand in candidates:
        try:
            if cand.is_dir:
                # Tar into a tmp file alongside the run dir to avoid
                # filling /tmp on the GPU host (which is often small).
                with tempfile.TemporaryDirectory(dir=str(run_dir)) as td:
                    tar_path = Path(td) / f"{cand.path.name}.tar.gz"
                    _tar_directory(cand.path, tar_path)
                    handle = await store.upload_file(
                        tar_path,
                        project_id=project_id,
                        kind=cand.kind,
                        job_id=job_id,
                        lineage=cand.lineage,
                    )
            else:
                handle = await store.upload_file(
                    cand.path,
                    project_id=project_id,
                    kind=cand.kind,
                    job_id=job_id,
                    lineage=cand.lineage,
                )
            logger.info("published %s -> %s", cand.relpath, handle.id)
            successes.append(handle)
        except (ArtifactError, OSError) as exc:
            logger.warning("publish failed for %s: %s", cand.relpath, exc)
            failures.append((cand.relpath, str(exc)))

    result = PublishResult(artifacts=successes, failed=failures)
    # Write manifest into run dir; the client picks it up via the
    # progress-sync rsync include list.
    manifest_path = run_dir / "artifacts.json"
    manifest_path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lqh.remote.publish",
        description="Publish a finished run's artifacts to api.lqh.ai (R2).",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--project-id",
        required=True,
        help="Project identifier (typically the lqh project directory name).",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Optional job UUID to associate artifacts with (cloud runs).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("LQH_BASE_URL"),
        help="API base URL (defaults to LQH_BASE_URL or https://api.lqh.ai/v1).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("LQH_API_TOKEN") or os.environ.get("LQH_DEBUG_API_KEY"),
        help="Bearer token (defaults to $LQH_API_TOKEN or $LQH_DEBUG_API_KEY).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.token is None:
        print(
            "publish: no auth token available — set LQH_API_TOKEN in the remote env",
            file=sys.stderr,
        )
        return 2

    result = asyncio.run(
        publish_run(
            args.run_dir,
            project_id=args.project_id,
            job_id=args.job_id,
            api_base=args.api_base,
            token=args.token,
        )
    )
    print(
        f"published {len(result.artifacts)} artifacts, "
        f"{len(result.failed)} failed",
        file=sys.stderr,
    )
    # Surface a non-zero exit if anything failed so the launcher's
    # caller can decide whether to mark the run partially successful.
    return 0 if not result.failed else 1


if __name__ == "__main__":
    sys.exit(main())
